-- Durable, ordered turn lifecycle events. The turn row is serialized before
-- every insert/update, so MAX(sequence)+1 is safe for transitions on that turn.
CREATE FUNCTION basalt_log_turn_transition() RETURNS TRIGGER LANGUAGE plpgsql AS $$
DECLARE
  next_seq BIGINT;
BEGIN
  SELECT COALESCE(MAX(sequence),0)+1 INTO next_seq FROM thread_events WHERE turn_id=NEW.id;
  INSERT INTO thread_events(turn_id,sequence,event_type,payload)
    VALUES(NEW.id,next_seq,'turn.'||lower(NEW.status),
           jsonb_build_object('status',NEW.status,'message_id',NEW.committed_message_id));
  RETURN NEW;
END $$;
CREATE TRIGGER basalt_turn_accepted_event AFTER INSERT ON turns FOR EACH ROW
  EXECUTE FUNCTION basalt_log_turn_transition();
CREATE TRIGGER basalt_turn_status_event AFTER UPDATE OF status ON turns
  FOR EACH ROW WHEN (OLD.status IS DISTINCT FROM NEW.status)
  EXECUTE FUNCTION basalt_log_turn_transition();
